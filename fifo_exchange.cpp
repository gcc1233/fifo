#define BOOST_TEST_MODULE ExchangeTests
#include <boost/test/included/unit_test.hpp>

// concepts
#include <optional>
#include <ranges>
#include <cstdint>
#include <concepts>


namespace gg::concepts {

// could put these in connectivity::concepts
template<typename T>
concept Depth = std::ranges::random_access_range<T> &&
                requires (T& t){
                    { t.empty() } -> std::same_as<bool>;
                    { t.full() } -> std::same_as<bool>;
                    { t.size() } -> std::same_as<std::size_t>;
                    t.push_back(std::declval<std::ranges::range_value_t<T>>());
                    { t.back() } -> std::same_as<std::remove_cvref_t<std::ranges::range_value_t<T>>&>;
                };

template<typename T>
concept Level = Depth<typename T::DepthType> &&
                requires {
                    typename T::EntryType;
                } &&
                std::same_as<std::remove_cvref_t<decltype(T::maxEntries)>, std::size_t>;

template<typename T>
concept LevelNode = Level<T> &&
                    std::same_as<std::remove_cvref_t<decltype(T::numLevels)>, std::size_t>;


template<typename T>
concept IndexedBookSide = requires (T&& t) {
    []<template<typename...>typename U, typename... V>(U<V...>&&){}.operator()(std::forward<T>(t));
} && requires (T& t) {
    { t.insert(1.0, 1) } -> std::same_as<std::optional<std::size_t>>;
    t.erase(std::size_t{});
};

template<typename T, typename ValueType>
concept RangeOf = std::ranges::range<T> && std::same_as<std::ranges::range_value_t<T>, ValueType>;

} // gg::concepts
// concepts

// meta
#include <boost/mp11/algorithm.hpp>
#include <boost/circular_buffer.hpp>

#include <concepts>
#include <memory>
#include <iterator>
#include <cstdint>


namespace gg::meta {

// can do this with partial specialization as well (single L<T...>)
template<typename T, template<typename...>typename U>
concept IsInstantiationOf = std::same_as<boost::mp11::mp_rename<T, U>, T>;

template<typename T>
struct Distance
{
    static void backDistance(T const& buff){ static_assert(false, "meta::Distance not implemented for this type"); }
    static void atDistance(T& buff, std::size_t distance){ static_assert(false, "meta::Distance not implemented for this type"); }   
};

template<IsInstantiationOf<boost::circular_buffer> T>
struct Distance<T> 
{
    // to avoid index invalidation by pop_front
    static std::size_t backDistance(T const& buff)
    {
        return static_cast<std::size_t>(std::distance(std::to_address(buff.array_two().first), std::addressof(buff.back())));
    }
    static auto& atDistance(T& buff, std::size_t distance)
    {
        return *(buff.array_two().first + distance);
    }
};

} // gg::meta
// meta

#include <cstdint>
#include <concepts>

namespace gg::math {

struct LevelFixedDecimal
{
    explicit constexpr LevelFixedDecimal(std::floating_point auto value)
    : value{static_cast<std::int64_t>(value * 100.0)}
    {}

    explicit constexpr LevelFixedDecimal(std::integral auto value)
    : value{static_cast<std::int64_t>(value)}
    {}

    constexpr LevelFixedDecimal operator+(LevelFixedDecimal const& other) const
    {
        return LevelFixedDecimal{value + other.value};
    }

    constexpr LevelFixedDecimal operator-(LevelFixedDecimal const& other) const
    {
        return LevelFixedDecimal{value - other.value};
    }

    constexpr LevelFixedDecimal operator/(LevelFixedDecimal const& other) const
    {
        return LevelFixedDecimal{value / other.value};
    }

    constexpr double toFloat() const { return static_cast<double>(value) / 100.0; }
    constexpr std::int64_t toInt() const { return value; }
    constexpr std::size_t toUint() const { return static_cast<std::uint64_t>(value); }

    std::int64_t value;
};

} // gg::math

// // container
#include <algorithm>
#include <vector>
#include <memory>
#include <numeric>
#include <cstdint>
#include <iterator>


namespace gg::container {

template<typename DataType>
struct MovableKeyMapRange
{
    using KeyType = std::uint32_t;

    explicit MovableKeyMapRange(std::size_t capacity)
    : keyToData(capacity, nullptr)
    , freeKeys(capacity)
    {
        data.reserve(capacity);
        endEnabled = data.begin();
        std::iota(freeKeys.rbegin(), freeKeys.rend(), KeyType{0});
    }

    MovableKeyMapRange(MovableKeyMapRange const&) = delete;
    MovableKeyMapRange& operator=(MovableKeyMapRange const&) = delete;

    MovableKeyMapRange(MovableKeyMapRange&&) = default;
    MovableKeyMapRange& operator=(MovableKeyMapRange&&) = default;

    template<typename... Args>
    [[nodiscard]] KeyType emplace(Args&&... args)
    {
        if (freeKeys.empty()) [[unlikely]]
            throw std::out_of_range("Range full, increase size");
    
        auto const nextKey = freeKeys.back();
        freeKeys.pop_back();
        auto& dataPtr = keyToData[nextKey];
        data.emplace_back(DataType{std::forward<Args>(args)...}, &dataPtr);
        dataPtr = &data.back();
        return nextKey; 
    }

    void disable(KeyType key)
    {
        if (keyToData[key] < std::to_address(endEnabled))
        {
            --endEnabled;
            std::swap(*endEnabled, *keyToData[key]);
        }
    }

    void enable(KeyType key)
    {
        if (keyToData[key] >= std::to_address(endEnabled))
        {
            std::swap(*keyToData[key], *endEnabled);
            ++endEnabled;
        }
    }

    void erase(KeyType key)
    {
        disable(key);
        std::swap(data.back(), *keyToData[key]);
        auto const numEnabled = enabled();
        data.pop_back();
        endEnabled = data.begin() + numEnabled;
        keyToData[key] = nullptr;
        freeKeys.push_back(key);
    }

    DataType& at(KeyType key){ return keyToData[key]->data; }
    DataType const& at(KeyType key) const { return keyToData[key]->data; }
    DataType& front(){ return data.front().data; }
    DataType const& front() const { return data.front().data; }

    bool isEnabled(KeyType key) const { return keyToData[key] < std::to_address(endEnabled); }
    
    template<typename Compare>
    void sortEnabled(Compare compare)
    {
        std::sort(data.begin(), endEnabled, [&](auto const& a, auto const& b){ return compare(a, b); });
    }

    template<typename Compare>
    void sortEnabledData(Compare compare)
    {
        std::sort(data.begin(), endEnabled, [&](auto const& a, auto const& b){ return compare(a.data, b.data); });
    }

    template<typename F>
    void forEnabled(F&& f)
    {
        for (auto it = data.begin(); it != endEnabled;)
        {
            auto const prevEnd = endEnabled;
            f(it->data);
            it += !(endEnabled < prevEnd);
        }
    }


    std::size_t enabled() const { return std::distance(data.begin(), typename DataStore::const_iterator(endEnabled)); }
    std::size_t size() const { return data.size(); }
    std::size_t capacity() const { return keyToData.size(); }

    struct DataCrossPtr
    {
        explicit DataCrossPtr(DataType&& data, DataCrossPtr** crossPtr)
        : data{std::move(data)}
        , crossPtr{crossPtr}
        {}

        DataCrossPtr(DataCrossPtr const&) = delete;
        DataCrossPtr& operator=(DataCrossPtr const&) = delete;

        DataCrossPtr(DataCrossPtr&& other)
        : data{std::move(other.data)}
        , crossPtr{other.crossPtr}
        {
            *crossPtr = this;
            other.crossPtr = nullptr;
        }

        DataCrossPtr& operator=(DataCrossPtr&& other)
        {
            if (&other != this)
            {
                data = std::move(other.data);
                crossPtr = other.crossPtr;
                *crossPtr = this;
                other.crossPtr = nullptr;
            }
            return *this;
        }

        KeyType key(MovableKeyMapRange* range) const
        {
            return static_cast<KeyType>(std::distance(std::to_address(range->keyToData.begin()), crossPtr));
        }

        DataType data;
    private:
        DataCrossPtr** crossPtr;
    };

    using DataStore = std::vector<DataCrossPtr>;

    DataStore data;
    std::vector<DataCrossPtr*> keyToData;
    std::vector<std::size_t> freeKeys;
    typename DataStore::iterator endEnabled;
};

} // gg::container
// container

// connectivity
// #include "gg/exchange/container.hpp"
// #include "gg/exchange/concepts.hpp"

#include <boost/circular_buffer.hpp>
#include <ranges>
#include <array>
#include <optional>
#include <functional>
#include <utility>
#include <cassert>
#include <cstdint>


namespace gg::connectivity {

template<typename EntryType>
using CircularDepth = boost::circular_buffer<EntryType>;

template<typename DepthType>
using DepthEntryType = std::ranges::range_value_t<DepthType>;

struct PriceVolumeEntry
{ 
    PriceVolumeEntry()
    {}

    PriceVolumeEntry(double price, int volume)
    : price{price}
    , volume{volume}
    , enabled{true}
    {}

    double price{};
    int volume{}; 
    bool enabled{false};
    constexpr void enable(){ enabled = true; } 
    constexpr void disable(){ enabled = false; } 
    constexpr bool isEnabled() const { return enabled; }
};

using PriceVolumeDepth = CircularDepth<PriceVolumeEntry>;

static_assert(concepts::Depth<PriceVolumeDepth>, "PriceVolumeDepth does not conform to Depth concept");


template<concepts::Depth DepthType_, std::size_t capacity_, std::size_t numLevels_>
struct LevelNode
{
    using DepthType = DepthType_;
    using EntryType = DepthEntryType<DepthType>;

    static constexpr std::size_t maxEntries = capacity_;
    static constexpr std::size_t numLevels = numLevels_;

    explicit LevelNode(std::size_t capacity, std::size_t prev = numLevels, std::size_t next = numLevels)
    : depth{capacity}
    , prev{prev}
    , next{next}
    {}

    constexpr bool empty() const
    {
        return depth.empty();
    }

    constexpr std::size_t size() const
    {
        return depth.size();
    }

    constexpr bool full() const
    {
        return depth.full();
    }

    DepthType depth;
    std::size_t prev;
    std::size_t next;
};

template<std::size_t capacity, std::size_t numLevels>
using PriceVolumeLevelNode = LevelNode<PriceVolumeDepth, capacity, numLevels>;

static_assert(concepts::LevelNode<PriceVolumeLevelNode<10, 10>>, "PriceVolumeLevelNode does not conform to LevelNode");


template<concepts::Depth DepthType_, std::size_t maxEntries_>
struct PriceVolumeLevel
{
    using DepthType = DepthType_;
    using EntryType = DepthEntryType<DepthType>; 

    static constexpr std::size_t maxEntries = maxEntries_;

    DepthType depth{maxEntries};
    double price{};
    int totalVolume{};
};

static_assert(concepts::Level<PriceVolumeLevel<PriceVolumeDepth, 10>>, "PriceVolumeLevel does not conform to Level concept");


// please be forgiving with this one; it isn't quite done yet :(
template<concepts::LevelNode LevelNodeType_>
struct LinkedRangeBookSide
{
    using LevelNodeType = LevelNodeType_;
    using DepthType = typename LevelNodeType::DepthType;
    using EntryType = typename LevelNodeType::EntryType;

    static constexpr std::size_t numLevels = LevelNodeType::numLevels;
    static constexpr std::size_t maxEntriesPerLevel = LevelNodeType::maxEntries;
    static constexpr std::size_t maxEntries = maxEntriesPerLevel * numLevels;

    using Entries = std::array<LevelNodeType, numLevels>;


    [[nodiscard]] std::optional<std::size_t> insert(double price, int volume)
    {
        assert(price > 0.0);
        assert(volume > 0);
        auto const roundedLevel = static_cast<std::size_t>(price);
        assert(roundedLevel < numLevels);
        auto& levelEntriesNode = entries[roundedLevel];
        if (levelEntriesNode.empty()) [[unlikely]]
        {
            findSetNeighbors(roundedLevel);
        }

        auto& depth = levelEntriesNode.depth; 
        if (!depth.full()) [[likely]]
        {
            depth.push_back(EntryType{price, volume});
            auto const index = backDistance(depth);
            auto const fullIndex = roundedLevel * maxEntriesPerLevel + index;
            return fullIndex;
        }
        else
        {
            return std::nullopt;
        }
    }

    void erase(std::size_t){};

    std::optional<std::reference_wrapper<EntryType>> at(std::size_t index)
    {
        auto const depthIndex = index % maxEntriesPerLevel;
        auto const levelIndex = (index - depthIndex) / maxEntriesPerLevel;
        auto& depth = entries[levelIndex].depth;
        return atDistance(depth, depthIndex);
    }

private:
    void findSetNeighbors(std::size_t idx)
    {
        auto const findPrevNext = [&]<int dir> 
        requires (dir == 0 || dir == 1) 
        (std::integral_constant<int, dir>)
        {
            auto [begin, end] = [&]
            { 
                if constexpr (dir == 0)
                {
                    return std::make_pair(this->entries.begin(), std::next(this->entries.begin(), idx));
                }
                else
                {
                    return std::make_pair(std::next(this->entries.begin(), idx + 1), this->entries.end());
                } 
            }();

            auto const it = std::find_if(
                begin, 
                end, 
                [&](auto const& level)
                { 
                    return !level.empty(); 
                }
            );

            if (it != end) [[likely]]
            {
                std::get<dir>(std::tie(it->next, it->prev)) = idx;
                auto const nbrIndex = static_cast<std::size_t>(std::distance(this->entries.begin(), it));
                std::get<dir>(std::tie(this->entries[idx].prev, this->entries[idx].next)) = nbrIndex;
            }
        };

        findPrevNext(std::integral_constant<int, 0>{});
        findPrevNext(std::integral_constant<int, 1>{});
    }

    static std::size_t backDistance(auto const& buff)
    {
        return static_cast<std::size_t>(std::distance(std::to_address(buff.array_two().first), std::addressof(buff.back())));
    }
    static auto& atDistance(auto& buff, std::size_t distance)
    {
        return *(buff.array_two().first + distance);
    }

    Entries entries{makeEntries()};

    static constexpr Entries makeEntries()
    {
        return [&]<std::size_t... levelIdx>(std::index_sequence<levelIdx...>)
        {
            return Entries{(levelIdx, LevelNodeType{ maxEntriesPerLevel })...};
        }(std::make_index_sequence<numLevels>{});
    }
};

static_assert(concepts::IndexedBookSide<LinkedRangeBookSide<PriceVolumeLevelNode<10, 10>>>, "LinkedRangeBookSide does not conform to IndexedBookSide concept");


template<concepts::Level LevelType>
struct DenseLevelsRangeBookSide : container::MovableKeyMapRange<LevelType>
{
    using Base = container::MovableKeyMapRange<LevelType>;
    using DepthType = typename LevelType::DepthType;
    using EntryType = typename LevelType::EntryType;

    static constexpr std::size_t maxEntriesPerLevel = LevelType::maxEntries;

    explicit DenseLevelsRangeBookSide(double startLevel_, double endLevel_, double tickSize_, unsigned side)
    : Base{((math::LevelFixedDecimal{endLevel_} - math::LevelFixedDecimal{startLevel_}) / math::LevelFixedDecimal{tickSize_}).toUint()}
    , startLevel{startLevel_}
    , endLevel{endLevel_}
    , tickSize{tickSize_}
    , side{side}
    {
        if (endLevel_ <= startLevel_)
        {
            throw std::range_error("endLevel must be larger than startLevel");
        }

        // using regular allocation of keys to give Key{0} -> Level{0 + start}; ex: for levels 100 -> 110, 100 has key = 0, 101 has key = 1
        // and these remain the same as long as no levels are removed (only disabled and enabled) 
        for (auto i = 0u; i < this->capacity(); ++i)
        {
            auto const key = this->emplace();
            this->at(key).price = (static_cast<double>(i) * tickSize.toFloat()) + startLevel.toFloat();
        }

        compare = side == 0u ? 
            std::function<bool(typename Base::DataCrossPtr const&, typename Base::DataCrossPtr const&)>{[&](auto const& a, auto const& b){ return a.key(this) > b.key(this); }} :
            std::function<bool(typename Base::DataCrossPtr const&, typename Base::DataCrossPtr const&)>{[&](auto const& a, auto const& b){ return a.key(this) < b.key(this); }};
    }

    // for ease of use, but maybe not needed or keep private
    using Base::at;
    using Base::capacity;
    using Base::disable;
    using Base::emplace;
    using Base::enable;
    using Base::enabled;
    using Base::forEnabled;
    using Base::front;
    using Base::isEnabled;
    using Base::sortEnabled;

    std::optional<std::size_t> insert(double price, int volume)
    {
        auto const roundedLevel = ((math::LevelFixedDecimal{price} - startLevel) / tickSize).toUint();
        if (!this->isEnabled(roundedLevel))
        {
            this->enable(roundedLevel);
            sortEnabledDataByKey();
        }

        auto& data = this->at(roundedLevel);
        auto& depth = data.depth;
        if (!depth.full()) [[likely]]
        {
            depth.push_back(EntryType{price, volume});
            data.totalVolume += volume;
            auto const dist = meta::Distance<DepthType>::backDistance(depth);
            auto const fullIndex = roundedLevel * maxEntriesPerLevel + dist;
            return fullIndex;
        }
        return std::nullopt;
    }

    bool erase(std::size_t index)
    {
        auto const depthIndex = index % maxEntriesPerLevel;
        auto const levelIndex = (index - depthIndex) / maxEntriesPerLevel;
        
        if (levelIndex >= this->capacity() || !this->isEnabled(levelIndex)) [[unlikely]]
            return false;

        auto& data = this->at(levelIndex);
        auto& depth = data.depth;
        auto& entry = meta::Distance<DepthType>::atDistance(depth, depthIndex);
        auto const volume = entry.volume;
        entry.disable();
        data.totalVolume -= volume;
        // would make exchangeOrderIds non-unique even before open
        // but leaving it also means getting top level takes more time (while (empty()){ std::prev })
        // we might solve this by requiring client to add clientId to delete order
        if (data.totalVolume <= 0)
        {
            this->disable(levelIndex); 
            sortEnabledDataByKey();
        }
        return true;
    }

    EntryType& entryAt(std::size_t index)
    {
        auto const depthIndex = index % maxEntriesPerLevel;
        auto const levelIndex = (index - depthIndex) / maxEntriesPerLevel;
        auto& data = this->at(levelIndex);
        auto& depth = data.depth;
        auto& entry = meta::Distance<DepthType>::atDistance(depth, depthIndex);
        return entry;
    }

private:
    // maybe not entirely ideal, can rrewrite to do sorted insert and delete with shift all ge/le
    void sortEnabledDataByKey(){ this->sortEnabled(compare); }

    math::LevelFixedDecimal startLevel;
    math::LevelFixedDecimal endLevel;
    math::LevelFixedDecimal tickSize;
    unsigned side;

    std::function<bool(typename Base::DataCrossPtr const&, typename Base::DataCrossPtr const&)> compare;
};

static_assert(concepts::IndexedBookSide<DenseLevelsRangeBookSide<PriceVolumeLevel<PriceVolumeDepth, 10>>>, "DenseLevelsRangeBookSide does not conform to IndexedBookSide concept");


struct TopLevelSide
{
    double price{};
    int volume{};

    bool operator<=>(TopLevelSide const&) const = default;
};

struct TopLevel
{
    using SideType = TopLevelSide;

    bool operator<=>(TopLevel const&) const = default;

    std::array<TopLevelSide, 2> bidAsk{};
};

} // gg::connectivity


// Iexchange
#include <functional>
#include <string>
#include <ostream>


// namespace gg::exchange {
// Keeping everything in global namespace for ease of use

// Do NOT modify this file or your code will not compile against gg's version.

enum class Side 
{ 
    Buy, 
    Sell 
};

using Price = double;
using Volume = int;
using UserReference = int;
using OrderId = int;

enum class InsertError 
{ 
    OK, 
    SymbolNotFound, 
    InvalidPrice, 
    InvalidVolume, 
    SystemError 
};

enum class DeleteError 
{ 
    OK, 
    OrderNotFound, 
    SystemError 
};


class IExchange
{
public:
    virtual ~IExchange() {}
    virtual void InsertOrder(
        const std::string& symbol,
        Side side,
        Price price,
        Volume volume,
        UserReference userReference
    ) = 0;

    virtual void DeleteOrder(
        OrderId orderId
    ) = 0;
    
    using OrderInsertedFunction = std::function<void (
        UserReference, 
        InsertError, 
        OrderId)
    >;
    OrderInsertedFunction OnOrderInserted;
    
    using OrderDeletedFunction = std::function<void (OrderId, DeleteError)>;
    OrderDeletedFunction OnOrderDeleted;
    
    using BestPriceChangedFunction = std::function<void (
        const std::string& symbol,
        Price bestBid,
        Volume totalBidVolume,
        Price bestAsk,
        Volume totalAskVolume)
    >;
    BestPriceChangedFunction OnBestPriceChanged;  
};

// can do this with BOOST_HANA_DEFINE/DESCRIBE_ENUM and enumDescribe (I'll put a little example in traits)
// I added one missing that might cause runtime_error
inline std::ostream& operator<<(std::ostream& os, InsertError er)
{
    switch (er)
    {
        case InsertError::OK:
            return os << "OK";
        case InsertError::SymbolNotFound:
            return os << "SymbolNotFound";
        case InsertError::InvalidPrice:
            return os << "InvalidPrice";
        case InsertError::InvalidVolume:
            return os << "InvalidVolume";
        case InsertError::SystemError:
            return os << "SystemError";
        default:
            throw std::runtime_error("Unhandled enum");
    }
}
// I change one broken output here on case DeleteError::OrderNotFound ("OK" -> "OrderNotFound")
inline std::ostream& operator<<(std::ostream& os, DeleteError er)
{
    switch (er)
    {
        case DeleteError::OK:
            return os << "OK";
        case DeleteError::OrderNotFound:
            return os << "OrderNotFound";
        case DeleteError::SystemError:
            return os << "SystemError";
        default:
            throw std::runtime_error("Unhandled enum");
    }
}

// } // gg::exchange

// FifoExchange
// #pragma once

// #include "gg/exchange/iexchange.hpp"

// #include "gg/exchange/connectivity.hpp"
// #include "gg/exchange/math.hpp"
// #include "gg/exchange/concepts.hpp"

#include <boost/unordered/unordered_flat_map.hpp>

#include <expected>
#include <ranges>
#include <array>
#include <utility>
#include <cstdint>


namespace gg::exchange {

// Note: keeping OrderId as int limits size of depth to 2^19 -> ~500,000
// (using 12 bits for symbol index -> 4,096)
template<
    math::LevelFixedDecimal startLevel_ = math::LevelFixedDecimal{0.0}, 
    math::LevelFixedDecimal endLevel_ = math::LevelFixedDecimal{200.0}, 
    math::LevelFixedDecimal tickSize_ = math::LevelFixedDecimal{1.0},
    std::size_t maxPerLevel = 100u
>
class FifoExchangeAbstract : public IExchange
{
public:
    static constexpr std::size_t maxSymbols = 10u;
    // make these dynamic in cstr
    static constexpr double startLevel = startLevel_.toFloat();
    static constexpr double endLevel = endLevel_.toFloat();
    static constexpr double tickSize = tickSize_.toFloat();
    static constexpr std::size_t numLevels = ((endLevel_ - startLevel_) / tickSize_).toUint();

    static constexpr std::size_t maxEntriesPerLevel = maxPerLevel;
    static constexpr std::size_t maxEntries = numLevels * maxEntriesPerLevel;
    // need to adjust logic to avoid using this index
    static constexpr int nullOrderId = -1;
    
    using SymbolType = std::string;
    using SymbolsList = std::array<SymbolType, maxSymbols>;
    using SymbolToIndex = boost::unordered_flat_map<SymbolType, unsigned>;

    using DepthType = connectivity::PriceVolumeDepth;
    using LevelType = connectivity::PriceVolumeLevel<DepthType, maxEntriesPerLevel>;

    using BookSideType = connectivity::DenseLevelsRangeBookSide<LevelType>;
    using BookType = std::array<BookSideType, 2>;

    using SymbolBooks = std::array<BookType, maxSymbols>;

    using OrderIdToClientId = boost::unordered_flat_map<OrderId, UserReference>;

    using TopLevels = std::array<connectivity::TopLevel, maxSymbols>;

    ~FifoExchangeAbstract() = default;
    FifoExchangeAbstract(concepts::RangeOf<SymbolType> auto const& symbols)
    {
        if (std::ranges::size(symbols) >= maxSymbols)
            throw std::out_of_range("too many symbols, increase max symbols");
        
        std::size_t symbolIndex = 0u;
        for (auto const& symbol : symbols)
        {
            auto res [[maybe_unused]] = symbolToIndex.try_emplace(symbol, symbolIndex++);
        }
        std::ranges::copy(symbols, symbolsList.begin());
    }

    void InsertOrder(
        SymbolType const& symbol,
        Side side,
        Price price,
        Volume volume,
        UserReference userReference
    ) override
    {
        auto symbolIndex = validateSymbolPriceVolume(symbol, price, volume);
        if (!symbolIndex.has_value()) [[unlikely]]
        {
            std::invoke(this->OnOrderInserted, userReference, symbolIndex.error(), nullOrderId);
            return;
        }
        auto exchangeOrderId = tryInsertOrder(*symbolIndex, side, price, volume);
        if (!exchangeOrderId.has_value()) [[unlikely]]
        {
            std::invoke(this->OnOrderInserted, userReference, exchangeOrderId.error(), nullOrderId);
            return;
        }
        orderIdToClientId[*exchangeOrderId] = userReference;
        std::invoke(this->OnOrderInserted, userReference, InsertError::OK, *exchangeOrderId);
        checkBBOChanged(*symbolIndex);
    }

    void DeleteOrder(
        OrderId orderId
    ) override
    {
        auto idcs = validateExchangeOrderId(orderId);
        if (!idcs.has_value()) [[unlikely]]
        {
            std::invoke(this->OnOrderDeleted, orderId, idcs.error());
            return;
        }
        auto [symbolIndex, side, orderIndex] = *idcs;
        auto const sideIndex = std::to_underlying(side);
        auto& bookSide = symbolBooks[symbolIndex][sideIndex];
        auto erased = bookSide.erase(orderIndex);
        if (!erased) [[unlikely]]
        {
            std::invoke(this->OnOrderDeleted, orderId, DeleteError::SystemError);
            return;
        }
        // need to fix to some null value or use map
        orderIdToClientId[orderId] = 0;
        std::invoke(this->OnOrderDeleted, orderId, DeleteError::OK);
        checkBBOChanged(symbolIndex);
    }

    void checkBBOChanged(std::size_t symbolIndex)
    {
        using TopLevelType = typename std::ranges::range_value_t<TopLevels>;
        using TopLevelSideType = typename TopLevelType::SideType;

        auto const& book = symbolBooks[symbolIndex];
        auto& topLevel = topLevels[symbolIndex];
        auto const prevTopLevel = std::exchange(
            topLevel, 
            TopLevelType{
                book[0].enabled() > 0u ? TopLevelSideType{book[0].front().price, book[0].front().totalVolume} : TopLevelSideType{},
                book[1].enabled() > 0u ? TopLevelSideType{book[1].front().price, book[1].front().totalVolume} : TopLevelSideType{}
            }
        );
        
        if (prevTopLevel != topLevel)
        {
            auto const& symbol = symbolsList[symbolIndex];
            auto const& [topLevelBid, topLevelAsk] = topLevel.bidAsk;
            std::invoke(this->OnBestPriceChanged, symbol, topLevelBid.price, topLevelBid.volume, topLevelAsk.price, topLevelAsk.volume);
        }
    }

    SymbolBooks symbolBooks{makeSymbolBooks()};
    SymbolToIndex symbolToIndex{maxSymbols};
    SymbolsList symbolsList{};
    OrderIdToClientId orderIdToClientId{maxSymbols};
    TopLevels topLevels{};

    using BookLocationId = std::tuple<std::size_t, Side, std::size_t>;

    std::expected<std::size_t, InsertError> validateSymbolPriceVolume(auto const& symbol, double price, int volume)
    {
        auto const symbolIt = symbolToIndex.find(symbol);
        if (symbolIt == symbolToIndex.end()) [[unlikely]]
            return std::unexpected(InsertError::SymbolNotFound);
        if (volume <= 0) [[unlikely]]
            return std::unexpected{InsertError::InvalidVolume};
        if (price < startLevel || price >= startLevel + tickSize * numLevels) [[unlikely]]
            return std::unexpected(InsertError::InvalidPrice);
        auto const symbolIndex = symbolIt->second;
        return symbolIndex;
    }

    std::expected<OrderId, InsertError> tryInsertOrder(std::size_t symbolIndex, Side side, double price, int volume)
    {
        auto const sideIndex = std::to_underlying(side);
        auto& bookSide = symbolBooks[symbolIndex][sideIndex];
        auto const orderIndex = bookSide.insert(price, volume);
        if (!orderIndex.has_value()) [[unlikely]]
        {
            return std::unexpected(InsertError::SystemError);
        }
        auto const exchangeOrderId = toExchangeOrderId(symbolIndex, side, *orderIndex);
        return exchangeOrderId;
    }

    std::expected<BookLocationId, DeleteError> validateExchangeOrderId(OrderId orderId)
    {
        auto const locationId = fromExchangeOrderId(orderId);
        auto const [symbolIndex, side, orderIndex] = locationId; 
        if (symbolIndex >= maxSymbols) [[unlikely]]
            return std::unexpected(DeleteError::OrderNotFound);
        auto const sideIndex = std::to_underlying(side);
        auto& bookSide = symbolBooks[symbolIndex][sideIndex];
        auto& entry = bookSide.entryAt(orderIndex);
        if (!entry.isEnabled()) [[unlikely]]
        {
            return std::unexpected(DeleteError::OrderNotFound);
        }
        return locationId;
    }


    static constexpr int symbolIdxBitSize = 12;
    static constexpr int orderIdxBitSize = 19;
    // arithmetic with n >= 0
    static constexpr int symbolIndexMask = std::numeric_limits<int>::max() >> (sizeof(int)*8 - (symbolIdxBitSize + 1));
    static constexpr int orderIndexMask = std::numeric_limits<int>::max() >> (sizeof(int)*8 - (orderIdxBitSize + 1));
    static constexpr OrderId toExchangeOrderId(std::size_t symbolIndex, Side side, std::size_t orderIndex)
    {
        // ideally OrderId would be unsigned; leaving it unchanged not to interfere with compilation
        OrderId result{};
        result |= (static_cast<int>(symbolIndex) & symbolIndexMask);
        result <<= orderIdxBitSize;
        result |= (static_cast<int>(orderIndex) & orderIndexMask);
        // I know; this is why I don't like doing bit stuff with int
        result |= std::bit_cast<unsigned>(std::to_underlying(side)) << (sizeof(unsigned)*8 - 1);
        return result;
    }
    static constexpr BookLocationId fromExchangeOrderId(OrderId orderId)
    {
        auto sideBit = (orderId >> (sizeof(int)*8 - 1)) * -1;
        auto const side = Side{sideBit};
        auto const orderIndex = static_cast<std::size_t>(orderId & orderIndexMask);
        auto const symbolIndex = static_cast<std::size_t>((orderId >> orderIdxBitSize) & symbolIndexMask);
        return { symbolIndex, side, orderIndex };
    }  

    // I prefer this to avoid excessive compiler messages from comment
    template<std::size_t, typename T>
    using IdentitySizeT = T;

    static constexpr SymbolBooks makeSymbolBooks()
    {
        return [&](auto iseq)
        {
            return [&]<std::size_t... sideIdx, std::size_t... symbolIdx>(std::index_sequence<sideIdx...>, std::index_sequence<symbolIdx...>)
            {
                    return SymbolBooks{IdentitySizeT<symbolIdx, BookType>{BookSideType{startLevel, endLevel, tickSize, static_cast<unsigned>(sideIdx)}...}...};
            }(std::make_index_sequence<2>{}, iseq);
        }(std::make_index_sequence<maxSymbols>{});
    }
};

using FifoExchange = FifoExchangeAbstract<>;


} // gg::exchange


// Testing
// #include "gg/exchange/fifo_exchange.hpp"
#include <boost/range/adaptor/indexed.hpp>
#include <set>

#include <memory>
#include <string>
// #include <boost/test/unit_test.hpp> 


using namespace std::string_literals;

namespace gg::exchange::testing {

struct FifoExchangeTestingFixture
{
    using ExchangeImplType = FifoExchange;
    using SymbolType = typename ExchangeImplType::SymbolType;

    struct InsertOrderResponse
    {
        UserReference userReference;
        OrderId orderId;
    };

    struct DeleteOrderResponse
    {
        OrderId orderId;
    };

    using BestPriceChangedResponse = connectivity::TopLevel;

    using InsertOrderResponses = std::vector<InsertOrderResponse>;
    using InsertResponsesByError = std::unordered_map<InsertError, InsertOrderResponses>;

    using DeleteOrderResponses = std::vector<DeleteOrderResponse>;
    using DeleteResponsesByError = std::unordered_map<DeleteError, DeleteOrderResponses>;

    using BestPriceChangedNotifications = std::vector<BestPriceChangedResponse>;
    using BestPriceChangedBySymbol = std::unordered_map<SymbolType, BestPriceChangedNotifications>;



    void OnOrderInserted(UserReference userReference, InsertError insertError, OrderId orderId)
    {
        insertResponses[insertError].emplace_back(userReference, orderId);
    }

    void OnOrderDeleted(OrderId orderId, DeleteError deleteError)
    {
        deleteResponses[deleteError].emplace_back(orderId);
    }
    
    void OnBestPriceChanged(SymbolType const& symbol, double bidPrice, int bidVolume, double askPrice, int askVolume)
    {
        bestPriceChanges[symbol].push_back(BestPriceChangedResponse{{{{bidPrice, bidVolume}, {askPrice, askVolume}}}});
    }

    template<typename ErrorType>
    std::size_t countResponses(ErrorType error)
    {
        if constexpr (std::same_as<ErrorType, InsertError>)
            return insertResponses[error].size();
        else if constexpr (std::same_as<ErrorType, DeleteError>)
            return deleteResponses[error].size();
        else
            return std::numeric_limits<std::size_t>::max();
    }

    const std::vector<SymbolType> symbols{
        "AABL"s,
        "AAPL"s,
        "GOOG"s,
        "MSFT"s
    };

    const std::vector<UserReference> userReferences{
        []
        { 
            auto v = std::vector<UserReference>(100); 
            std::iota(v.begin(), v.end(), 1); 
            return v; 
        }()
    };

    const std::vector<typename FifoExchange::SymbolType> badSymbols{
        ""s,
        "aapl"s,
        "NONE"s,
        "a"s
    };

    const std::vector<int> badVolumes{0, -10};
    const std::vector<double> badPrices{-10.0, FifoExchange::endLevel + 1.0};

    InsertResponsesByError insertResponses{};
    DeleteResponsesByError deleteResponses{};
    BestPriceChangedBySymbol bestPriceChanges{};

    FifoExchangeTestingFixture()
    : exchange{std::make_unique<FifoExchange>(symbols)}
    {
        using std::placeholders::_1;
        using std::placeholders::_2;
        using std::placeholders::_3;
        using std::placeholders::_4;
        using std::placeholders::_5;

        exchange->OnOrderInserted = std::bind(&FifoExchangeTestingFixture::OnOrderInserted, this, _1, _2, _3);
        exchange->OnOrderDeleted = std::bind(&FifoExchangeTestingFixture::OnOrderDeleted, this, _1, _2);
        exchange->OnBestPriceChanged = std::bind(&FifoExchangeTestingFixture::OnBestPriceChanged, this, _1, _2, _3, _4, _5);
    }

    ~FifoExchangeTestingFixture(){}

    std::unique_ptr<IExchange> exchange;
};


BOOST_FIXTURE_TEST_SUITE(FifoExchangeTests, FifoExchangeTestingFixture)


BOOST_AUTO_TEST_CASE(InsertOrderInsertErrorSymbolNotFound)
{
    for (auto const& symbol : badSymbols)
        exchange->InsertOrder(symbol, Side::Buy, 1.0, 1, 1);
    
    BOOST_CHECK_EQUAL(countResponses(InsertError::SymbolNotFound), badSymbols.size());  
    BOOST_CHECK_EQUAL(countResponses(InsertError::OK), 0u);         
    BOOST_CHECK_EQUAL(countResponses(InsertError::InvalidVolume), 0u);
    BOOST_CHECK_EQUAL(countResponses(InsertError::InvalidPrice), 0u);
    BOOST_CHECK_EQUAL(countResponses(InsertError::SystemError), 0u);
}

BOOST_AUTO_TEST_CASE(InsertOrderInsertErrorInvalidVolume)
{
    for (auto&& [v, clientId] : std::ranges::views::zip(badVolumes, userReferences))
        exchange->InsertOrder(symbols[0], Side::Buy, 1.0, v, clientId);
    
    BOOST_CHECK_EQUAL(countResponses(InsertError::InvalidVolume), badVolumes.size());   
    BOOST_CHECK_EQUAL(countResponses(InsertError::OK), 0u);   
    BOOST_CHECK_EQUAL(countResponses(InsertError::SymbolNotFound), 0u);      
    BOOST_CHECK_EQUAL(countResponses(InsertError::InvalidPrice), 0u);
    BOOST_CHECK_EQUAL(countResponses(InsertError::SystemError), 0u);
}

BOOST_AUTO_TEST_CASE(InsertOrderInsertErrorInvalidPrice)
{
    for (auto&& [p, clientId] : std::ranges::views::zip(badPrices, userReferences))
        exchange->InsertOrder(symbols[0], Side::Buy, p, 1, clientId);
    
    BOOST_CHECK_EQUAL(countResponses(InsertError::InvalidPrice), badPrices.size());   
    BOOST_CHECK_EQUAL(countResponses(InsertError::OK), 0u);   
    BOOST_CHECK_EQUAL(countResponses(InsertError::SymbolNotFound), 0u);      
    BOOST_CHECK_EQUAL(countResponses(InsertError::InvalidVolume), 0u);
    BOOST_CHECK_EQUAL(countResponses(InsertError::SystemError), 0u);
}

BOOST_AUTO_TEST_CASE(InsertOrderInsertErrorSystemError)
{
    for (auto i = 0u; i <= FifoExchange::maxEntriesPerLevel; ++i)
        exchange->InsertOrder(symbols[0], Side::Buy, 1.0, 1, 1);
    
    BOOST_CHECK_EQUAL(countResponses(InsertError::SystemError), 1u);   
    BOOST_CHECK_EQUAL(countResponses(InsertError::OK), FifoExchange::maxEntriesPerLevel);   
    BOOST_CHECK_EQUAL(countResponses(InsertError::SymbolNotFound), 0u);      
    BOOST_CHECK_EQUAL(countResponses(InsertError::InvalidVolume), 0u);
    BOOST_CHECK_EQUAL(countResponses(InsertError::InvalidPrice), 0u);
}

BOOST_AUTO_TEST_CASE(InsertOrderOKCorrectClientIdBuyAndSellAndUniqueOrderId)
{
    std::set<OrderId> orderIds;
    for (auto&& [symbol, clientId] : std::ranges::views::zip(symbols, userReferences))
    {
        auto& inserted = insertResponses[InsertError::OK];
        for (auto price = FifoExchange::startLevel; price < FifoExchange::endLevel; price += FifoExchange::tickSize + 0.001)
        {
            auto const prevSize = inserted.size();
            
            exchange->InsertOrder(symbol, Side::Buy, price, 1, clientId);
            exchange->InsertOrder(symbol, Side::Sell, price, 1, clientId);
            
            BOOST_TEST_REQUIRE(inserted.size() == prevSize + 2u);

            auto const& first = inserted[inserted.size() - 2];
            auto const& second = inserted.back();
            BOOST_CHECK_EQUAL(first.userReference, clientId);
            auto [it_f, inserted_f] = orderIds.insert(first.orderId);
            BOOST_TEST(inserted_f);
            BOOST_CHECK_EQUAL(second.userReference, clientId);
            auto [it_s, inserted_s] = orderIds.insert(second.orderId);
            BOOST_TEST(inserted_s);
        }
    }
}

BOOST_AUTO_TEST_CASE(DeleteOrderDeleteErrorOrderNotFoundNoOrderAndOKAndNotFoundDeletedOrder)
{
    auto const nonOrders = std::vector{1, 2, 3, 4};

    // should have done this above too
    // john always loved to do this
    // though he would do it this way:
    // auto& deletedNotFound = deleteResponses[DeleteError::OrderNotFound];
    // auto const prevSize = deletedNotFound.size();
    auto const prevSizeNotFound = countResponses(DeleteError::OrderNotFound);

    for (auto const orderId : nonOrders)
        exchange->DeleteOrder(orderId);

    BOOST_CHECK_EQUAL(countResponses(DeleteError::OrderNotFound), prevSizeNotFound + std::ranges::size(nonOrders));   
    BOOST_CHECK_EQUAL(countResponses(DeleteError::OK), 0u);   
    BOOST_CHECK_EQUAL(countResponses(DeleteError::SystemError), 0u);

    auto const prevSizeInserted = countResponses(InsertError::OK);

    for (auto i = 0u; i < 4u; ++i)
        exchange->InsertOrder(symbols[0], Side::Buy, 2.0, 1, userReferences[i]);

    BOOST_TEST_REQUIRE(countResponses(InsertError::OK) == prevSizeInserted + 4u);     

    auto const activeOrderIds = insertResponses[InsertError::OK] | std::ranges::views::all | std::ranges::views::transform([&](auto const& insertResponse){ return insertResponse.orderId; });
    
    auto const prevDeletedOK = countResponses(DeleteError::OK);

    for (auto const& orderId : activeOrderIds)
        exchange->DeleteOrder(orderId);
    
    BOOST_CHECK_EQUAL(countResponses(DeleteError::OK), prevDeletedOK + 4u);   
    BOOST_CHECK_EQUAL(countResponses(DeleteError::OrderNotFound), std::ranges::size(nonOrders));   
    BOOST_CHECK_EQUAL(countResponses(DeleteError::SystemError), 0u);

    // nicer like this
    auto const secondPrevSizeNotFound = countResponses(DeleteError::OrderNotFound);

    for (auto const& orderId : activeOrderIds)
        exchange->DeleteOrder(orderId);
    
    BOOST_CHECK_EQUAL(countResponses(DeleteError::OK), prevDeletedOK + 4u);   
    BOOST_CHECK_EQUAL(countResponses(DeleteError::OrderNotFound), secondPrevSizeNotFound + 4u);   
    BOOST_CHECK_EQUAL(countResponses(DeleteError::SystemError), 0u);
}

BOOST_AUTO_TEST_CASE(DeleteOrderDeleteErrorSystemError)
{
}

BOOST_AUTO_TEST_CASE(DeleteOrderDeleteErrorOKDiffSymbolsLevelsSides)
{
    auto& inserted = insertResponses[InsertError::OK];
    for (auto const& symbol : symbols)
    {
        for (auto price : {2.0, 8.0})
        {
            auto const prevSize = inserted.size();
            
            exchange->InsertOrder(symbol, Side::Buy, price, 1, 1);
            exchange->InsertOrder(symbol, Side::Sell, price, 1, 1);
            
            BOOST_CHECK_EQUAL(inserted.size(), prevSize + 2u);
        }
    }

    auto& deletedOK = deleteResponses[DeleteError::OK];
    for (auto const& res : inserted)
    {
        auto const orderId = res.orderId;
        auto const prevSize = deletedOK.size();
        exchange->DeleteOrder(orderId);
        BOOST_CHECK_EQUAL(deletedOK.size(), prevSize + 1);
        BOOST_CHECK_EQUAL(deletedOK.back().orderId, orderId);
    }
}

BOOST_AUTO_TEST_CASE(BBOCorrectOnInsertNewLevelPerSideIncreaseVolumeAndRemoveLevel)
{
    // Insert new level 
    for (auto const& symbol : symbols)
    {
        exchange->InsertOrder(symbol, Side::Buy, 2.0, 1, 1);
        auto const& topLevel = bestPriceChanges[symbol].back();
        auto const& [bidLevel, askLevel] = topLevel.bidAsk;
        BOOST_CHECK_CLOSE(bidLevel.price, 2.0, 0.001);
        BOOST_CHECK_EQUAL(bidLevel.volume, 1);
        BOOST_CHECK_CLOSE(askLevel.price, 0.0, 0.001);
        BOOST_CHECK_EQUAL(askLevel.volume, 0);
    } 

    // Add side
    for (auto const& symbol : symbols)
    {
        exchange->InsertOrder(symbol, Side::Sell, 4.0, 1, 1);
        auto const& topLevel = bestPriceChanges[symbol].back();
        auto const& [bidLevel, askLevel] = topLevel.bidAsk;
        BOOST_CHECK_CLOSE(bidLevel.price, 2.0, 0.001);
        BOOST_CHECK_EQUAL(bidLevel.volume, 1);
        BOOST_CHECK_CLOSE(askLevel.price, 4.0, 0.001);
        BOOST_CHECK_EQUAL(askLevel.volume, 1);
    }   
    
    // Increase Volume
    for (auto const& symbol : symbols)
    {
        exchange->InsertOrder(symbol, Side::Buy, 2.0, 1, 1);
        auto const& topLevel = bestPriceChanges[symbol].back();
        auto const& [bidLevel, askLevel] = topLevel.bidAsk;
        BOOST_CHECK_CLOSE(bidLevel.price, 2.0, 0.001);
        BOOST_CHECK_EQUAL(bidLevel.volume, 2);
        BOOST_CHECK_CLOSE(askLevel.price, 4.0, 0.001);
        BOOST_CHECK_EQUAL(askLevel.volume, 1);
    }   

    // note tick size setting

    std::vector<OrderId> topLevelOrderIds{};

    // Add new higher level
    for (auto const& symbol : symbols)
    {
        exchange->InsertOrder(symbol, Side::Buy, 3.001, 1, 1);
        auto const& topLevel = bestPriceChanges[symbol].back();
        auto const& [bidLevel, askLevel] = topLevel.bidAsk;
        BOOST_CHECK_CLOSE(bidLevel.price, 3.0, 0.001);
        BOOST_CHECK_EQUAL(bidLevel.volume, 1);
        BOOST_CHECK_CLOSE(askLevel.price, 4.0, 0.001);
        BOOST_CHECK_EQUAL(askLevel.volume, 1);
        auto const orderId = insertResponses[InsertError::OK].back().orderId;
        topLevelOrderIds.push_back(orderId);
    }  

    // Remove level gives prev
    for (auto&& [index, symbol] : symbols | boost::adaptors::indexed(0))
    {
        exchange->DeleteOrder(topLevelOrderIds[index]);
        auto const& topLevel = bestPriceChanges[symbol].back();
        auto const& [bidLevel, askLevel] = topLevel.bidAsk;
        BOOST_CHECK_CLOSE(bidLevel.price, 2.0, 0.001);
        BOOST_CHECK_EQUAL(bidLevel.volume, 2);
        BOOST_CHECK_CLOSE(askLevel.price, 4.0, 0.001);
        BOOST_CHECK_EQUAL(askLevel.volume, 1);
    }  
}

BOOST_AUTO_TEST_SUITE_END()

} // gg::exchange::testing